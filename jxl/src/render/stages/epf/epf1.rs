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

#[inline(always)]
fn epf1_process_row_chunk_impl<D: SimdDescriptor>(
    d: D,
    stage: &Epf1Stage,
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

    let sm = stage.sigma_scale * 1.65;
    let bsm = sm * stage.border_sad_mul;
    let sad_mul_storage = prepare_sad_mul_storage(xpos, ypos, sm, bsm);

    let inputs: [[Window<4>; 5]; 3] =
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

            let p22_x = in_chunks[0][2].get::<2>();
            let p22_y = in_chunks[1][2].get::<2>();
            let p22_b = in_chunks[2][2].get::<2>();

            let sigma_mask = D::F32Vec::splat(d, MIN_SIGMA).gt(sigma);
            if sigma_mask.all() {
                out_chunks[0].write(p22_x);
                out_chunks[1].write(p22_y);
                out_chunks[2].write(p22_b);
                return;
            }

            let inv_sigma = sigma * sad_mul;

            let mut w_acc = D::F32Vec::splat(d, 1.0);
            let mut out_x = p22_x;
            let mut out_y = p22_y;
            let mut out_b = p22_b;

            // Pair 1: Vertical neighbors: North (p21, sad0) and South (p23, sad3)
            let mut sad0 = D::F32Vec::splat(d, 0.0);
            let mut sad3 = D::F32Vec::splat(d, 0.0);
            for (c, &scale) in stage.channel_scale.iter().enumerate() {
                let scale = D::F32Vec::splat(d, scale);
                let p21 = in_chunks[c][1].get::<2>();
                let p22 = in_chunks[c][2].get::<2>();
                let p23 = in_chunks[c][3].get::<2>();

                let d22_21 = (p22 - p21).abs();
                let d22_23 = (p22 - p23).abs();
                let shared = d22_21 + d22_23;

                let p20 = in_chunks[c][0].get::<2>();
                let p11 = in_chunks[c][1].get::<1>();
                let p12 = in_chunks[c][2].get::<1>();
                let p31 = in_chunks[c][1].get::<3>();
                let p32 = in_chunks[c][2].get::<3>();
                let diff0 = shared + (p20 - p21).abs() + (p11 - p12).abs() + (p31 - p32).abs();

                let p24 = in_chunks[c][4].get::<2>();
                let p13 = in_chunks[c][3].get::<1>();
                let p33 = in_chunks[c][3].get::<3>();
                let diff3 = shared + (p24 - p23).abs() + (p13 - p12).abs() + (p33 - p32).abs();

                sad0 = diff0.mul_add(scale, sad0);
                sad3 = diff3.mul_add(scale, sad3);
            }

            let w0 = sad0
                .mul_add(inv_sigma, D::F32Vec::splat(d, 1.0))
                .max(D::F32Vec::splat(d, 0.0));
            let w3 = sad3
                .mul_add(inv_sigma, D::F32Vec::splat(d, 1.0))
                .max(D::F32Vec::splat(d, 0.0));
            w_acc += w0 + w3;
            out_x = w0.mul_add(
                in_chunks[0][1].get::<2>(),
                w3.mul_add(in_chunks[0][3].get::<2>(), out_x),
            );
            out_y = w0.mul_add(
                in_chunks[1][1].get::<2>(),
                w3.mul_add(in_chunks[1][3].get::<2>(), out_y),
            );
            out_b = w0.mul_add(
                in_chunks[2][1].get::<2>(),
                w3.mul_add(in_chunks[2][3].get::<2>(), out_b),
            );

            // Pair 2: Horizontal neighbors: West (p12, sad1) and East (p32, sad2)
            let mut sad1 = D::F32Vec::splat(d, 0.0);
            let mut sad2 = D::F32Vec::splat(d, 0.0);
            for (c, &scale) in stage.channel_scale.iter().enumerate() {
                let scale = D::F32Vec::splat(d, scale);
                let p12 = in_chunks[c][2].get::<1>();
                let p22 = in_chunks[c][2].get::<2>();
                let p32 = in_chunks[c][2].get::<3>();

                let d12_22 = (p22 - p12).abs();
                let d22_32 = (p22 - p32).abs();
                let shared = d12_22 + d22_32;

                let p02 = in_chunks[c][2].get::<0>();
                let p11 = in_chunks[c][1].get::<1>();
                let p21 = in_chunks[c][1].get::<2>();
                let p13 = in_chunks[c][3].get::<1>();
                let p23 = in_chunks[c][3].get::<2>();
                let diff1 = shared + (p02 - p12).abs() + (p11 - p21).abs() + (p13 - p23).abs();

                let p42 = in_chunks[c][2].get::<4>();
                let p31 = in_chunks[c][1].get::<3>();
                let p33 = in_chunks[c][3].get::<3>();
                let diff2 = shared + (p42 - p32).abs() + (p31 - p21).abs() + (p33 - p23).abs();

                sad1 = diff1.mul_add(scale, sad1);
                sad2 = diff2.mul_add(scale, sad2);
            }

            let w1 = sad1
                .mul_add(inv_sigma, D::F32Vec::splat(d, 1.0))
                .max(D::F32Vec::splat(d, 0.0));
            let w2 = sad2
                .mul_add(inv_sigma, D::F32Vec::splat(d, 1.0))
                .max(D::F32Vec::splat(d, 0.0));
            w_acc += w1 + w2;
            out_x = w1.mul_add(
                in_chunks[0][2].get::<1>(),
                w2.mul_add(in_chunks[0][2].get::<3>(), out_x),
            );
            out_y = w1.mul_add(
                in_chunks[1][2].get::<1>(),
                w2.mul_add(in_chunks[1][2].get::<3>(), out_y),
            );
            out_b = w1.mul_add(
                in_chunks[2][2].get::<1>(),
                w2.mul_add(in_chunks[2][2].get::<3>(), out_b),
            );

            let inv_w = D::F32Vec::splat(d, 1.0) / w_acc;
            out_x *= inv_w;
            out_y *= inv_w;
            out_b *= inv_w;
            out_x = sigma_mask.if_then_else_f32(p22_x, out_x);
            out_y = sigma_mask.if_then_else_f32(p22_y, out_y);
            out_b = sigma_mask.if_then_else_f32(p22_b, out_b);
            out_chunks[0].write(out_x);
            out_chunks[1].write(out_y);
            out_chunks[2].write(out_b);
        },
    );
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
        epf1_process_row_chunk_impl(d, stage, pos, xsize, input_rows, output_rows)
    }
);

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

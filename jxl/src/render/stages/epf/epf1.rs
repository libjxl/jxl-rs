// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use jxl_simd::{F32SimdVec, SimdDescriptor, SimdMask, simd_function};

use crate::features::epf::SigmaSource;
use crate::render::stages::epf::common::{
    accumulate_pair, get_sigma, prepare_sad_mul_storage,
};
use crate::render::{
    Channels, ChannelsMut, ErasedLocalState, ForEachChunk, RenderPipelineInOutStage,
};
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
fn epf1_process_simd<D: SimdDescriptor>(
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

    let sigma = stage.sigma.try_read().unwrap();
    let row_sigma = sigma.row(ypos / BLOCK_DIM);

    let sm = stage.sigma_scale * 1.65;
    let bsm = sm * stage.border_sad_mul;
    let sad_mul_storage = prepare_sad_mul_storage(xpos, ypos, sm, bsm);

    let scale0 = D::F32Vec::splat(d, stage.channel_scale[0]);
    let scale1 = D::F32Vec::splat(d, stage.channel_scale[1]);
    let scale2 = D::F32Vec::splat(d, stage.channel_scale[2]);
    let c_one = D::F32Vec::splat(d, 1.0);
    let c_zero = D::F32Vec::splat(d, 0.0);
    let min_sigma = D::F32Vec::splat(d, MIN_SIGMA);

    ForEachChunk::<3, 5, 2, 3, 1>::run(
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
                out_view.store::<0, 0>(in_view.load::<0, 2, 0>());
                out_view.store::<1, 0>(in_view.load::<1, 2, 0>());
                out_view.store::<2, 0>(in_view.load::<2, 2, 0>());
                return;
            }

            let inv_sigma = sigma * sad_mul;

            let mut w_acc = c_one;
            let mut out0 = in_view.load::<0, 2, 0>();
            let mut out1 = in_view.load::<1, 2, 0>();
            let mut out2 = in_view.load::<2, 2, 0>();

            // Pair 1: Vertical neighbors: North (p21, sad0) and South (p23, sad3)
            let mut sad0 = c_zero;
            let mut sad3 = c_zero;
            macro_rules! vertical_channel {
                ($c:expr, $scale:expr) => {{
                    let scale = $scale;
                    let p21 = in_view.load::<$c, 1, 0>();
                    let p22 = in_view.load::<$c, 2, 0>();
                    let p23 = in_view.load::<$c, 3, 0>();

                    let d22_21 = (p22 - p21).abs();
                    let d22_23 = (p22 - p23).abs();
                    let shared = d22_21 + d22_23;

                    let p20 = in_view.load::<$c, 0, 0>();
                    let p11 = in_view.load::<$c, 1, -1>();
                    let p12 = in_view.load::<$c, 2, -1>();
                    let p31 = in_view.load::<$c, 1, 1>();
                    let p32 = in_view.load::<$c, 2, 1>();
                    let diff0 = shared + (p20 - p21).abs() + (p11 - p12).abs() + (p31 - p32).abs();

                    let p24 = in_view.load::<$c, 4, 0>();
                    let p13 = in_view.load::<$c, 3, -1>();
                    let p33 = in_view.load::<$c, 3, 1>();
                    let diff3 = shared + (p24 - p23).abs() + (p13 - p12).abs() + (p33 - p32).abs();

                    sad0 = diff0.mul_add(scale, sad0);
                    sad3 = diff3.mul_add(scale, sad3);
                }};
            }
            vertical_channel!(0, scale0);
            vertical_channel!(1, scale1);
            vertical_channel!(2, scale2);

            let w0 = sad0.mul_add(inv_sigma, c_one).max(c_zero);
            let w3 = sad3.mul_add(inv_sigma, c_one).max(c_zero);
            w_acc += w0 + w3;

            accumulate_pair!(in_view, out0, out1, out2, w0, 1, 0, w3, 3, 0);

            // Pair 2: Horizontal neighbors: West (p12, sad1) and East (p32, sad2)
            let mut sad1 = c_zero;
            let mut sad2 = c_zero;
            macro_rules! horizontal_channel {
                ($c:expr, $scale:expr) => {{
                    let scale = $scale;
                    let p12 = in_view.load::<$c, 2, -1>();
                    let p22 = in_view.load::<$c, 2, 0>();
                    let p32 = in_view.load::<$c, 2, 1>();

                    let d12_22 = (p22 - p12).abs();
                    let d22_32 = (p22 - p32).abs();
                    let shared = d12_22 + d22_32;

                    let p02 = in_view.load::<$c, 2, -2>();
                    let p11 = in_view.load::<$c, 1, -1>();
                    let p21 = in_view.load::<$c, 1, 0>();
                    let p13 = in_view.load::<$c, 3, -1>();
                    let p23 = in_view.load::<$c, 3, 0>();
                    let diff1 = shared + (p02 - p12).abs() + (p11 - p21).abs() + (p13 - p23).abs();

                    let p42 = in_view.load::<$c, 2, 2>();
                    let p31 = in_view.load::<$c, 1, 1>();
                    let p33 = in_view.load::<$c, 3, 1>();
                    let diff2 = shared + (p42 - p32).abs() + (p31 - p21).abs() + (p33 - p23).abs();

                    sad1 = diff1.mul_add(scale, sad1);
                    sad2 = diff2.mul_add(scale, sad2);
                }};
            }
            horizontal_channel!(0, scale0);
            horizontal_channel!(1, scale1);
            horizontal_channel!(2, scale2);

            let w1 = sad1.mul_add(inv_sigma, c_one).max(c_zero);
            let w2 = sad2.mul_add(inv_sigma, c_one).max(c_zero);
            w_acc += w1 + w2;

            accumulate_pair!(in_view, out0, out1, out2, w1, 2, -1, w2, 2, 1);

            let inv_w = c_one / w_acc;
            out0 *= inv_w;
            out1 *= inv_w;
            out2 *= inv_w;

            out_view.store::<0, 0>(sigma_mask.if_then_else_f32(in_view.load::<0, 2, 0>(), out0));
            out_view.store::<1, 0>(sigma_mask.if_then_else_f32(in_view.load::<1, 2, 0>(), out1));
            out_view.store::<2, 0>(sigma_mask.if_then_else_f32(in_view.load::<2, 2, 0>(), out2));
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
        epf1_process_simd(d, stage, pos, xsize, input_rows, output_rows);
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

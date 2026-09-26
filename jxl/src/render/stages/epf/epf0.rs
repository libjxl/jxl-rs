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
fn epf0_process_simd<D: SimdDescriptor>(
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

    ForEachChunk::<3, 7, 3, 3, 1>::run(
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
                out_view.store::<0, 0>(in_view.load::<0, 3, 0>());
                out_view.store::<1, 0>(in_view.load::<1, 3, 0>());
                out_view.store::<2, 0>(in_view.load::<2, 3, 0>());
                return;
            }

            let inv_sigma = sigma * sad_mul;
            let mut w = c_one;
            let mut out0 = in_view.load::<0, 3, 0>();
            let mut out1 = in_view.load::<1, 3, 0>();
            let mut out2 = in_view.load::<2, 3, 0>();

            // Set 1: Vertical +-1 neighbors (sad2: row 2, col 0; sad9: row 4, col 0)
            {
                let mut sad2 = c_zero;
                let mut sad9 = c_zero;
                macro_rules! set1_channel {
                    ($c:expr, $scale:expr) => {{
                        let scale = $scale;
                        let p32 = in_view.load::<$c, 2, 0>();
                        let p23 = in_view.load::<$c, 3, -1>();
                        let p33 = in_view.load::<$c, 3, 0>();
                        let p43 = in_view.load::<$c, 3, 1>();
                        let p34 = in_view.load::<$c, 4, 0>();

                        let d23_22 = (p23 - in_view.load::<$c, 2, -1>()).abs();
                        let d33_32 = (p32 - p33).abs();
                        let d43_42 = (p43 - in_view.load::<$c, 2, 1>()).abs();
                        let d23_24 = (p23 - in_view.load::<$c, 4, -1>()).abs();
                        let d33_34 = (p34 - p33).abs();
                        let d43_44 = (p43 - in_view.load::<$c, 4, 1>()).abs();

                        let diffs_2 = (p32 - in_view.load::<$c, 1, 0>()).abs()
                            + d23_22
                            + d33_32
                            + d43_42
                            + d33_34;

                        let diffs_9 = d33_32
                            + d23_24
                            + d33_34
                            + d43_44
                            + (p34 - in_view.load::<$c, 5, 0>()).abs();

                        sad2 = scale.mul_add(diffs_2, sad2);
                        sad9 = scale.mul_add(diffs_9, sad9);
                    }};
                }
                set1_channel!(0, scale0);
                set1_channel!(1, scale1);
                set1_channel!(2, scale2);

                let w2 = sad2.mul_add(inv_sigma, c_one).max(c_zero);
                let w9 = sad9.mul_add(inv_sigma, c_one).max(c_zero);
                w += w2 + w9;

                accumulate_pair!(in_view, out0, out1, out2, w2, 2, 0, w9, 4, 0);
            }

            // Set 2: Vertical +-2 neighbors (sad0: row 1, col 0; sad11: row 5, col 0)
            {
                let mut sad0 = c_zero;
                let mut sad11 = c_zero;
                macro_rules! set2_channel {
                    ($c:expr, $scale:expr) => {{
                        let scale = $scale;
                        let p32 = in_view.load::<$c, 2, 0>();
                        let p34 = in_view.load::<$c, 4, 0>();

                        let d32_34 = (p32 - p34).abs();
                        let d23_21 =
                            (in_view.load::<$c, 3, -1>() - in_view.load::<$c, 1, -1>()).abs();
                        let d43_41 =
                            (in_view.load::<$c, 3, 1>() - in_view.load::<$c, 1, 1>()).abs();
                        let d23_25 =
                            (in_view.load::<$c, 3, -1>() - in_view.load::<$c, 5, -1>()).abs();
                        let d43_45 =
                            (in_view.load::<$c, 3, 1>() - in_view.load::<$c, 5, 1>()).abs();

                        let diffs_0 = (p32 - in_view.load::<$c, 0, 0>()).abs()
                            + d23_21
                            + (in_view.load::<$c, 3, 0>() - in_view.load::<$c, 1, 0>()).abs()
                            + d43_41
                            + d32_34;

                        let diffs_11 = d32_34
                            + d23_25
                            + (in_view.load::<$c, 3, 0>() - in_view.load::<$c, 5, 0>()).abs()
                            + d43_45
                            + (p34 - in_view.load::<$c, 6, 0>()).abs();

                        sad0 = scale.mul_add(diffs_0, sad0);
                        sad11 = scale.mul_add(diffs_11, sad11);
                    }};
                }
                set2_channel!(0, scale0);
                set2_channel!(1, scale1);
                set2_channel!(2, scale2);

                let w0 = sad0.mul_add(inv_sigma, c_one).max(c_zero);
                let w11 = sad11.mul_add(inv_sigma, c_one).max(c_zero);
                w += w0 + w11;

                accumulate_pair!(in_view, out0, out1, out2, w0, 1, 0, w11, 5, 0);
            }

            // Set 3: Horizontal +-1 neighbors (sad5: row 3, col -1; sad6: row 3, col 1)
            {
                let mut sad5 = c_zero;
                let mut sad6 = c_zero;
                macro_rules! set3_channel {
                    ($c:expr, $scale:expr) => {{
                        let scale = $scale;
                        let p32 = in_view.load::<$c, 2, 0>();
                        let p23 = in_view.load::<$c, 3, -1>();
                        let p33 = in_view.load::<$c, 3, 0>();
                        let p43 = in_view.load::<$c, 3, 1>();
                        let p34 = in_view.load::<$c, 4, 0>();

                        let d23_13 = (p23 - in_view.load::<$c, 3, -2>()).abs();
                        let d33_23 = (p33 - p23).abs();
                        let d33_43 = (p43 - p33).abs();
                        let d43_53 = (in_view.load::<$c, 3, 2>() - p43).abs();

                        let diffs_5 = (p32 - in_view.load::<$c, 2, -1>()).abs()
                            + d23_13
                            + d33_23
                            + d33_43
                            + (p34 - in_view.load::<$c, 4, -1>()).abs();

                        let diffs_6 = (p32 - in_view.load::<$c, 2, 1>()).abs()
                            + d33_23
                            + d33_43
                            + d43_53
                            + (p34 - in_view.load::<$c, 4, 1>()).abs();

                        sad5 = scale.mul_add(diffs_5, sad5);
                        sad6 = scale.mul_add(diffs_6, sad6);
                    }};
                }
                set3_channel!(0, scale0);
                set3_channel!(1, scale1);
                set3_channel!(2, scale2);

                let w5 = sad5.mul_add(inv_sigma, c_one).max(c_zero);
                let w6 = sad6.mul_add(inv_sigma, c_one).max(c_zero);
                w += w5 + w6;

                accumulate_pair!(in_view, out0, out1, out2, w5, 3, -1, w6, 3, 1);
            }

            // Set 4: Horizontal +-2 neighbors (sad4: row 3, col -2; sad7: row 3, col 2)
            {
                let mut sad4 = c_zero;
                let mut sad7 = c_zero;
                macro_rules! set4_channel {
                    ($c:expr, $scale:expr) => {{
                        let scale = $scale;
                        let p32 = in_view.load::<$c, 2, 0>();
                        let p23 = in_view.load::<$c, 3, -1>();
                        let p33 = in_view.load::<$c, 3, 0>();
                        let p43 = in_view.load::<$c, 3, 1>();
                        let p34 = in_view.load::<$c, 4, 0>();

                        let d23_43 = (p23 - p43).abs();

                        let diffs_4 = (p32 - in_view.load::<$c, 2, -2>()).abs()
                            + (p23 - in_view.load::<$c, 3, -3>()).abs()
                            + (p33 - in_view.load::<$c, 3, -2>()).abs()
                            + d23_43
                            + (p34 - in_view.load::<$c, 4, -2>()).abs();

                        let diffs_7 = (p32 - in_view.load::<$c, 2, 2>()).abs()
                            + d23_43
                            + (p33 - in_view.load::<$c, 3, 2>()).abs()
                            + (p43 - in_view.load::<$c, 3, 3>()).abs()
                            + (p34 - in_view.load::<$c, 4, 2>()).abs();

                        sad4 = scale.mul_add(diffs_4, sad4);
                        sad7 = scale.mul_add(diffs_7, sad7);
                    }};
                }
                set4_channel!(0, scale0);
                set4_channel!(1, scale1);
                set4_channel!(2, scale2);

                let w4 = sad4.mul_add(inv_sigma, c_one).max(c_zero);
                let w7 = sad7.mul_add(inv_sigma, c_one).max(c_zero);
                w += w4 + w7;

                accumulate_pair!(in_view, out0, out1, out2, w4, 3, -2, w7, 3, 2);
            }

            // Set 5: Diagonal Main \ neighbors (sad1: row 2, col -1; sad10: row 4, col 1)
            {
                let mut sad1 = c_zero;
                let mut sad10 = c_zero;
                macro_rules! set5_channel {
                    ($c:expr, $scale:expr) => {{
                        let scale = $scale;
                        let p32 = in_view.load::<$c, 2, 0>();
                        let p23 = in_view.load::<$c, 3, -1>();
                        let p33 = in_view.load::<$c, 3, 0>();
                        let p43 = in_view.load::<$c, 3, 1>();
                        let p34 = in_view.load::<$c, 4, 0>();

                        let d32_43 = (p32 - p43).abs();
                        let d23_34 = (p23 - p34).abs();
                        let shared = d32_43 + d23_34;

                        let diffs_1 = (p32 - in_view.load::<$c, 1, -1>()).abs()
                            + (p23 - in_view.load::<$c, 2, -2>()).abs()
                            + (p33 - in_view.load::<$c, 2, -1>()).abs()
                            + shared;

                        let diffs_10 = shared
                            + (p33 - in_view.load::<$c, 4, 1>()).abs()
                            + (p43 - in_view.load::<$c, 4, 2>()).abs()
                            + (p34 - in_view.load::<$c, 5, 1>()).abs();

                        sad1 = scale.mul_add(diffs_1, sad1);
                        sad10 = scale.mul_add(diffs_10, sad10);
                    }};
                }
                set5_channel!(0, scale0);
                set5_channel!(1, scale1);
                set5_channel!(2, scale2);

                let w1 = sad1.mul_add(inv_sigma, c_one).max(c_zero);
                let w10 = sad10.mul_add(inv_sigma, c_one).max(c_zero);
                w += w1 + w10;

                accumulate_pair!(in_view, out0, out1, out2, w1, 2, -1, w10, 4, 1);
            }

            // Set 6: Diagonal Anti / neighbors (sad3: row 2, col 1; sad8: row 4, col -1)
            {
                let mut sad3 = c_zero;
                let mut sad8 = c_zero;
                macro_rules! set6_channel {
                    ($c:expr, $scale:expr) => {{
                        let scale = $scale;
                        let p32 = in_view.load::<$c, 2, 0>();
                        let p23 = in_view.load::<$c, 3, -1>();
                        let p33 = in_view.load::<$c, 3, 0>();
                        let p43 = in_view.load::<$c, 3, 1>();
                        let p34 = in_view.load::<$c, 4, 0>();

                        let d32_23 = (p32 - p23).abs();
                        let d43_34 = (p43 - p34).abs();
                        let shared = d32_23 + d43_34;

                        let diffs_3 = (p32 - in_view.load::<$c, 1, 1>()).abs()
                            + (p33 - in_view.load::<$c, 2, 1>()).abs()
                            + (p43 - in_view.load::<$c, 2, 2>()).abs()
                            + shared;

                        let diffs_8 = shared
                            + (p23 - in_view.load::<$c, 4, -2>()).abs()
                            + (p33 - in_view.load::<$c, 4, -1>()).abs()
                            + (p34 - in_view.load::<$c, 5, -1>()).abs();

                        sad3 = scale.mul_add(diffs_3, sad3);
                        sad8 = scale.mul_add(diffs_8, sad8);
                    }};
                }
                set6_channel!(0, scale0);
                set6_channel!(1, scale1);
                set6_channel!(2, scale2);

                let w3 = sad3.mul_add(inv_sigma, c_one).max(c_zero);
                let w8 = sad8.mul_add(inv_sigma, c_one).max(c_zero);
                w += w3 + w8;

                accumulate_pair!(in_view, out0, out1, out2, w3, 2, 1, w8, 4, -1);
            }

            let inv_w = c_one / w;
            out0 *= inv_w;
            out1 *= inv_w;
            out2 *= inv_w;

            out_view.store::<0, 0>(sigma_mask.if_then_else_f32(in_view.load::<0, 3, 0>(), out0));
            out_view.store::<1, 0>(sigma_mask.if_then_else_f32(in_view.load::<1, 3, 0>(), out1));
            out_view.store::<2, 0>(sigma_mask.if_then_else_f32(in_view.load::<2, 3, 0>(), out2));
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
        epf0_process_simd(d, stage, pos, xsize, input_rows, output_rows);
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

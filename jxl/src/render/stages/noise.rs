// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#![allow(clippy::needless_range_loop)]

use jxl_simd::{F32SimdVec, I32SimdVec, SimdDescriptor, SimdMask, simd_function};

use crate::features::noise::Noise;
use crate::frame::color_correlation_map::ColorCorrelationParams;
use crate::render::{
    Channels, ChannelsMut, ErasedLocalState, RenderPipelineInOutStage, RenderPipelineInPlaceStage,
};
use crate::util::sync::{Arc, RwLock};

pub struct ConvolveNoiseStage {
    channel: usize,
}

impl ConvolveNoiseStage {
    pub fn new(channel: usize) -> ConvolveNoiseStage {
        ConvolveNoiseStage { channel }
    }
}

impl std::fmt::Display for ConvolveNoiseStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "convolve noise for channel {}", self.channel,)
    }
}

// SIMD noise convolution (5x5 kernel)
simd_function!(
    convolve_noise_simd_dispatch,
    d: D,
    fn convolve_noise_simd(input: &[&[f32]], output: &mut [f32], xsize: usize) {
        // Precompute constants
        let c016 = D::F32Vec::splat(d, 0.16);
        let cn384 = D::F32Vec::splat(d, -3.84);

        // Windows of size LEN+4 from each row (for offsets 0..5), stepping by LEN
        let iter0 = input[0].windows(D::F32Vec::LEN + 4).step_by(D::F32Vec::LEN);
        let iter1 = input[1].windows(D::F32Vec::LEN + 4).step_by(D::F32Vec::LEN);
        let iter2 = input[2].windows(D::F32Vec::LEN + 4).step_by(D::F32Vec::LEN);
        let iter3 = input[3].windows(D::F32Vec::LEN + 4).step_by(D::F32Vec::LEN);
        let iter4 = input[4].windows(D::F32Vec::LEN + 4).step_by(D::F32Vec::LEN);
        let out_iter = output.chunks_exact_mut(D::F32Vec::LEN);

        for ((((w0, w1), w2), w3), (w4, out)) in iter0
            .zip(iter1)
            .zip(iter2)
            .zip(iter3)
            .zip(iter4.zip(out_iter))
            .take(xsize.div_ceil(D::F32Vec::LEN))
        {
            // Load center pixel (row 2, offset +2)
            let p00 = D::F32Vec::load(d, &w2[2..]);

            // Accumulate surrounding pixels
            let mut others = D::F32Vec::splat(d, 0.0);

            // Add all 5 offsets for rows 0, 1, 3, 4
            for i in 0..5 {
                others += D::F32Vec::load(d, &w0[i..]);
                others += D::F32Vec::load(d, &w1[i..]);
                others += D::F32Vec::load(d, &w3[i..]);
                others += D::F32Vec::load(d, &w4[i..]);
            }

            // Add row 2 neighbors (skip center at offset 2)
            others += D::F32Vec::load(d, &w2[0..]);
            others += D::F32Vec::load(d, &w2[1..]);
            others += D::F32Vec::load(d, &w2[3..]);
            others += D::F32Vec::load(d, &w2[4..]);

            // Compute: others * 0.16 + center * -3.84
            let result = others.mul_add(c016, p00 * cn384);
            result.store(out);
        }
    }
);

impl RenderPipelineInOutStage for ConvolveNoiseStage {
    type InputT = f32;
    type OutputT = f32;
    const SHIFT: (u8, u8) = (0, 0);
    const BORDER: (u8, u8) = (2, 2);

    fn uses_channel(&self, c: usize) -> bool {
        c == self.channel
    }

    fn process_row_chunk(
        &self,
        _position: (usize, usize),
        xsize: usize,
        input_rows: &Channels<f32>,
        output_rows: &mut ChannelsMut<f32>,
        _state: Option<&mut ErasedLocalState>,
    ) {
        let input = &input_rows[0];
        convolve_noise_simd_dispatch(input, output_rows[0][0], xsize);
    }
}

#[inline(always)]
fn add_noise_simd_impl<D: SimdDescriptor>(
    d: D,
    row_c: &mut [&mut [f32]],
    row_rnd: &[&[f32]],
    lut: &[f32; 8],
    ytox: f32,
    ytob: f32,
    xsize: usize,
) {
    let table = D::F32Vec::prepare_table_bf16_8(d, lut);
    let c_zero = D::F32Vec::zero(d);
    let c_one = D::F32Vec::splat(d, 1.0);
    let c_six = D::F32Vec::splat(d, 6.0);
    let c_half = D::F32Vec::splat(d, 0.5);
    let c_norm = D::F32Vec::splat(d, 0.22);
    let c_rgn_corr = D::F32Vec::splat(d, 0.0078125);
    let c_rg_corr = D::F32Vec::splat(d, 0.9921875);
    let c_ytox = D::F32Vec::splat(d, ytox);
    let c_ytob = D::F32Vec::splat(d, ytob);
    let c_i32_one = D::I32Vec::splat(d, 1);

    let noise_strength = {
        #[inline(always)]
        |vx: D::F32Vec| -> D::F32Vec {
            let scaled_vx = (vx * c_six).max(c_zero);
            let pre_floor_x = scaled_vx.floor();
            let pre_frac_x = scaled_vx - pre_floor_x;
            let is_ge_7 = pre_floor_x.gt(c_six);
            let floor_x = pre_floor_x.min(c_six);
            let frac_x = is_ge_7.if_then_else_f32(c_one, pre_frac_x);
            let floor_x_int = floor_x.as_i32();
            let low = D::F32Vec::table_lookup_bf16_8(d, table, floor_x_int);
            let hi = D::F32Vec::table_lookup_bf16_8(d, table, floor_x_int + c_i32_one);
            (hi - low).mul_add(frac_x, low).max(c_zero).min(c_one)
        }
    };

    let (row_c0, rest_c) = row_c.split_at_mut(1);
    let (row_c1, row_c2) = rest_c.split_at_mut(1);

    let iter_c0 = row_c0[0].chunks_exact_mut(D::F32Vec::LEN);
    let iter_c1 = row_c1[0].chunks_exact_mut(D::F32Vec::LEN);
    let iter_c2 = row_c2[0].chunks_exact_mut(D::F32Vec::LEN);
    let iter_rnd_r = row_rnd[0].chunks_exact(D::F32Vec::LEN);
    let iter_rnd_g = row_rnd[1].chunks_exact(D::F32Vec::LEN);
    let iter_rnd_c = row_rnd[2].chunks_exact(D::F32Vec::LEN);

    for (((((c0, c1), c2), rnd_r_chunk), rnd_g_chunk), rnd_c_chunk) in iter_c0
        .zip(iter_c1)
        .zip(iter_c2)
        .zip(iter_rnd_r)
        .zip(iter_rnd_g)
        .zip(iter_rnd_c)
        .take(xsize.div_ceil(D::F32Vec::LEN))
    {
        let vx = D::F32Vec::load(d, c0);
        let vy = D::F32Vec::load(d, c1);
        let vb = D::F32Vec::load(d, c2);

        let in_g = (vy - vx) * c_half;
        let in_r = (vy + vx) * c_half;

        let noise_strength_g = noise_strength(in_g);
        let noise_strength_r = noise_strength(in_r);

        let rnd_r = D::F32Vec::load(d, rnd_r_chunk) * c_norm;
        let rnd_g = D::F32Vec::load(d, rnd_g_chunk) * c_norm;
        let rnd_c = D::F32Vec::load(d, rnd_c_chunk) * c_norm;

        let red_noise = noise_strength_r * (rnd_r.mul_add(c_rgn_corr, rnd_c * c_rg_corr));
        let green_noise = noise_strength_g * (rnd_g.mul_add(c_rgn_corr, rnd_c * c_rg_corr));
        let rg_noise = red_noise + green_noise;

        let out_x = vx + rg_noise.mul_add(c_ytox, red_noise - green_noise);
        let out_y = vy + rg_noise;
        let out_b = vb + rg_noise * c_ytob;

        out_x.store(c0);
        out_y.store(c1);
        out_b.store(c2);
    }
}

// SIMD noise addition
simd_function!(
    add_noise_simd_dispatch,
    d: D,
    fn add_noise_simd(
        row_c: &mut [&mut [f32]],
        row_rnd: &[&[f32]],
        lut: &[f32; 8],
        ytox: f32,
        ytob: f32,
        xsize: usize,
    ) {
        add_noise_simd_impl(d, row_c, row_rnd, lut, ytox, ytob, xsize)
    }
);

pub struct AddNoiseStage {
    noise: Arc<RwLock<Noise>>,
    first_channel: usize,
    color_correlation: Arc<RwLock<ColorCorrelationParams>>,
}

impl AddNoiseStage {
    #[allow(dead_code)]
    pub fn new(
        noise: Arc<RwLock<Noise>>,
        color_correlation: Arc<RwLock<ColorCorrelationParams>>,
        first_channel: usize,
    ) -> AddNoiseStage {
        assert!(first_channel > 2);
        AddNoiseStage {
            noise,
            first_channel,
            color_correlation,
        }
    }
}

impl std::fmt::Display for AddNoiseStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "add noise for channels [{},{},{}]",
            self.first_channel,
            self.first_channel + 1,
            self.first_channel + 2
        )
    }
}

impl RenderPipelineInPlaceStage for AddNoiseStage {
    type Type = f32;

    fn uses_channel(&self, c: usize) -> bool {
        c < 3 || (c >= self.first_channel && c < self.first_channel + 3)
    }

    fn process_row_chunk(
        &self,
        _position: (usize, usize),
        xsize: usize,
        row: &mut [&mut [f32]],
        _state: Option<&mut ErasedLocalState>,
    ) {
        let noise = self.noise.try_read().unwrap();
        if noise.lut == [0.0; 8] {
            return;
        }
        let color_correlation = self.color_correlation.try_read().unwrap();
        let ytox = color_correlation.y_to_x_lf();
        let ytob = color_correlation.y_to_b_lf();

        let (row_c, rest) = row.split_at_mut(3);
        let row_rnd: [&[f32]; 3] = [&*rest[0], &*rest[1], &*rest[2]];

        add_noise_simd_dispatch(row_c, &row_rnd, &noise.lut, ytox, ytob, xsize);
    }
}

#[cfg(test)]
mod test {
    use test_log::test;

    use crate::error::Result;
    use crate::features::noise::Noise;
    use crate::frame::color_correlation_map::ColorCorrelationParams;
    use crate::image::Image;
    use crate::render::stages::noise::{AddNoiseStage, ConvolveNoiseStage};
    use crate::render::test::make_and_run_simple_pipeline;
    use crate::tests::assert_close;
    use crate::util::sync::{Arc, RwLock};

    // TODO(firsching): Add more relevant ConvolveNoise tests as per discussions in https://github.com/libjxl/jxl-rs/pull/60.

    #[test]
    fn convolve_noise_process_row_chunk() -> Result<()> {
        let input: Image<f32> = Image::new_range((2, 2), 0.0, 1.0)?;
        let stage = ConvolveNoiseStage::new(0);
        let output: Vec<Image<f32>> =
            make_and_run_simple_pipeline(stage, &[input], (2, 2), 0, 256)?;
        assert_close!(output[0].row(0)[0], 7.2, 1e-6);
        assert_close!(output[0].row(0)[1], 2.4, 1e-6);
        assert_close!(output[0].row(1)[0], -2.4, 1e-6);
        assert_close!(output[0].row(1)[1], -7.2, 1e-6);
        Ok(())
    }

    #[test]
    fn convolve_noise_consistency() -> Result<()> {
        crate::render::test::test_stage_consistency(|| ConvolveNoiseStage::new(0), (500, 500), 1)
    }

    // TODO(firsching): Add more relevant AddNoise tests as per discussions in https://github.com/libjxl/jxl-rs/pull/60.

    #[test]
    fn add_noise_process_row_chunk() -> Result<()> {
        let xsize = 8;
        let ysize = 8;
        let input_c0: Image<f32> = Image::new_range((xsize, ysize), 0.1, 0.1)?;
        let input_c1: Image<f32> = Image::new_range((xsize, ysize), 0.1, 0.1)?;
        let input_c2: Image<f32> = Image::new_range((xsize, ysize), 0.1, 0.1)?;
        let input_c3: Image<f32> = Image::new_range((xsize, ysize), 0.1, 0.1)?;
        let input_c4: Image<f32> = Image::new_range((xsize, ysize), 0.1, 0.1)?;
        let input_c5: Image<f32> = Image::new_range((xsize, ysize), 0.1, 0.1)?;
        let stage = AddNoiseStage::new(
            Arc::new(RwLock::new(Noise {
                lut: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            })),
            Arc::new(RwLock::new(ColorCorrelationParams::default())),
            3,
        );
        let output = make_and_run_simple_pipeline(
            stage,
            &[input_c0, input_c1, input_c2, input_c3, input_c4, input_c5],
            (xsize, ysize),
            0,
            256,
        )?;
        // Golden data generated by libjxl.
        let want_out = [
            [
                [
                    0.100000, 0.200000, 0.300000, 0.400000, 0.500000, 0.600000, 0.700000, 0.800000,
                ],
                [0.900000, 1.000000, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
                [1.7, 1.8, 1.9, 2.000000, 2.1, 2.2, 2.3, 2.4],
                [
                    2.5, 2.6, 2.7, 2.799999, 2.899999, 2.999999, 3.099999, 3.199999,
                ],
                [
                    3.299999, 3.399999, 3.499999, 3.599999, 3.699999, 3.799999, 3.899998, 3.999998,
                ],
                [
                    4.099998, 4.199998, 4.299998, 4.399998, 4.499998, 4.599998, 4.699998, 4.799998,
                ],
                [
                    4.899998, 4.999998, 5.099998, 5.199997, 5.299997, 5.399997, 5.499997, 5.599997,
                ],
                [
                    5.699997, 5.799997, 5.899997, 5.999997, 6.099997, 6.199996, 6.299996, 6.399996,
                ],
            ],
            [
                [
                    0.144000, 0.288000, 0.432000, 0.576000, 0.720000, 0.864000, 1.008, 1.152,
                ],
                [1.296, 1.44, 1.584, 1.728, 1.872, 2.016, 2.16, 2.304],
                [2.448, 2.592, 2.736001, 2.88, 3.024, 3.168, 3.312, 3.456],
                [
                    3.6, 3.743999, 3.888, 4.031999, 4.175999, 4.319999, 4.463999, 4.607999,
                ],
                [
                    4.751998, 4.895998, 5.039998, 5.183998, 5.327998, 5.471998, 5.615998, 5.759997,
                ],
                [
                    5.903998, 6.047997, 6.191998, 6.335998, 6.479997, 6.623997, 6.767997, 6.911997,
                ],
                [
                    7.055997, 7.199996, 7.343997, 7.487996, 7.631996, 7.775996, 7.919996, 8.063995,
                ],
                [
                    8.207995, 8.351995, 8.495996, 8.639996, 8.783995, 8.927995, 9.071995, 9.215995,
                ],
            ],
            [
                [
                    0.144000, 0.288000, 0.432000, 0.576000, 0.720000, 0.864000, 1.008, 1.152,
                ],
                [1.296, 1.44, 1.584, 1.728, 1.872, 2.016, 2.16, 2.304],
                [2.448, 2.592, 2.736001, 2.88, 3.024, 3.168, 3.312, 3.456],
                [
                    3.6, 3.743999, 3.888, 4.031999, 4.175999, 4.319999, 4.463999, 4.607999,
                ],
                [
                    4.751998, 4.895998, 5.039998, 5.183998, 5.327998, 5.471998, 5.615998, 5.759997,
                ],
                [
                    5.903998, 6.047997, 6.191998, 6.335998, 6.479997, 6.623997, 6.767997, 6.911997,
                ],
                [
                    7.055997, 7.199996, 7.343997, 7.487996, 7.631996, 7.775996, 7.919996, 8.063995,
                ],
                [
                    8.207995, 8.351995, 8.495996, 8.639996, 8.783995, 8.927995, 9.071995, 9.215995,
                ],
            ],
        ];
        for c in 0..3 {
            for y in 0..output[c].size().1 {
                for x in 0..output[c].size().0 {
                    assert_close!(output[c].row(y)[x], want_out[c][y][x], 1e-5);
                }
            }
        }
        Ok(())
    }

    #[test]
    fn add_noise_consistency() -> Result<()> {
        crate::render::test::test_stage_consistency(
            || {
                AddNoiseStage::new(
                    Arc::new(RwLock::new(Noise {
                        lut: [0.0, 2.0, 1.0, 0.0, 1.0, 3.0, 1.1, 2.3],
                    })),
                    Arc::new(RwLock::new(ColorCorrelationParams::default())),
                    3,
                )
            },
            (500, 500),
            6,
        )
    }
}

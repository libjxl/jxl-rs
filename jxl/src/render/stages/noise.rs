// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#![allow(clippy::needless_range_loop)]

use crate::{
    features::noise::Noise,
    frame::color_correlation_map::ColorCorrelationParams,
    render::{RenderPipelineInOutStage, RenderPipelineInPlaceStage},
};
use jxl_simd::{F32SimdVec, simd_function};

pub struct ConvolveNoiseStage {
    channel: usize,
}

impl ConvolveNoiseStage {
    pub fn new(channel: usize) -> ConvolveNoiseStage {
        ConvolveNoiseStage { channel }
    }
}

// SIMD implementation matching C++ stage_noise.cc lines 272-289
simd_function!(
    convolve_noise_simd_dispatch,
    d: D,
    fn convolve_noise_simd(
        input: &[&[f32]],
        output: &mut [f32],
        xsize: usize,
    ) {
        let simd_width = D::F32Vec::LEN;

        // Precompute constants
        let c016 = D::F32Vec::splat(d, 0.16);
        let cn384 = D::F32Vec::splat(d, -3.84);

        // Process in SIMD chunks
        let mut x = 0;
        while x + simd_width <= xsize {
            // Load center pixel (row 2, offset +2)
            let p00 = D::F32Vec::load(d, &input[2][x + 2..]);

            // Accumulate surrounding pixels (matching C++ pattern)
            let mut others = D::F32Vec::splat(d, 0.0);

            // Add all 5 rows × 5 offsets (except center)
            for i in 0..5 {
                others = others + D::F32Vec::load(d, &input[0][x + i..]);
                others = others + D::F32Vec::load(d, &input[1][x + i..]);
                others = others + D::F32Vec::load(d, &input[3][x + i..]);
                others = others + D::F32Vec::load(d, &input[4][x + i..]);
            }

            // Add row 2 neighbors (offset 0, 1, 3, 4 - skip center at offset 2)
            others = others + D::F32Vec::load(d, &input[2][x..]);
            others = others + D::F32Vec::load(d, &input[2][x + 1..]);
            others = others + D::F32Vec::load(d, &input[2][x + 3..]);
            others = others + D::F32Vec::load(d, &input[2][x + 4..]);

            // Compute: others * 0.16 + center * -3.84
            let result = others.mul_add(c016, p00 * cn384);

            result.store(&mut output[x..]);
            x += simd_width;
        }

        // Scalar tail
        for x in x..xsize {
            let mut others = 0.0;
            for i in 0..5 {
                others += input[0][x + i];
                others += input[1][x + i];
                others += input[3][x + i];
                others += input[4][x + i];
            }
            others += input[2][x];
            others += input[2][x + 1];
            others += input[2][x + 3];
            others += input[2][x + 4];
            output[x] = others * 0.16 + input[2][x + 2] * -3.84;
        }
    }
);

impl std::fmt::Display for ConvolveNoiseStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "convolve noise for channel {}", self.channel,)
    }
}

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
        input_rows: &[&[&[f32]]],
        output_rows: &mut [&mut [&mut [f32]]],
        _state: Option<&mut dyn std::any::Any>,
    ) {
        let input = input_rows[0];
        convolve_noise_simd_dispatch(input, output_rows[0][0], xsize);
    }
}

pub struct AddNoiseStage {
    noise: Noise,
    first_channel: usize,
    color_correlation: ColorCorrelationParams,
}

impl AddNoiseStage {
    #[allow(dead_code)]
    pub fn new(
        noise: Noise,
        color_correlation: ColorCorrelationParams,
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

// SIMD implementation matching C++ stage_noise.cc lines 170-246
simd_function!(
    add_noise_simd_dispatch,
    d: D,
    fn add_noise_simd(
        row: &mut [&mut [f32]],
        xsize: usize,
        noise: &Noise,
        ytox: f32,
        ytob: f32,
    ) {
        let simd_width = D::F32Vec::LEN;

        let norm_const = D::F32Vec::splat(d, 0.22);
        let half = D::F32Vec::splat(d, 0.5);
        let k_rg_corr = D::F32Vec::splat(d, 0.9921875);
        let k_rgn_corr = D::F32Vec::splat(d, 0.0078125);
        let ytox_v = D::F32Vec::splat(d, ytox);
        let ytob_v = D::F32Vec::splat(d, ytob);

        // Process in SIMD chunks
        let mut x = 0;
        while x + simd_width <= xsize {
            // Load random noise channels
            let row_rnd_r = D::F32Vec::load(d, &row[3][x..]);
            let row_rnd_g = D::F32Vec::load(d, &row[4][x..]);
            let row_rnd_c = D::F32Vec::load(d, &row[5][x..]);

            // Load color channels
            let vx = D::F32Vec::load(d, &row[0][x..]);
            let vy = D::F32Vec::load(d, &row[1][x..]);

            // Compute in_g and in_r (matching C++ lines 211-212)
            let in_g = vy - vx;
            let in_r = vy + vx;

            // Compute noise strengths (matching C++ lines 213-214)
            let in_g_half = in_g * half;
            let in_r_half = in_r * half;

            // For SIMD, we need to process noise strength calculation
            // C++ uses NoiseStrength function with table lookup
            // For now, use scalar fallback for table lookup
            let mut noise_strength_g_arr = [0.0f32; 16]; // Max SIMD lanes
            let mut noise_strength_r_arr = [0.0f32; 16];

            // Store to temp arrays for scalar processing
            let mut in_g_half_arr = [0.0f32; 16];
            let mut in_r_half_arr = [0.0f32; 16];
            in_g_half.store(&mut in_g_half_arr);
            in_r_half.store(&mut in_r_half_arr);

            for i in 0..simd_width {
                noise_strength_g_arr[i] = noise.strength(in_g_half_arr[i]);
                noise_strength_r_arr[i] = noise.strength(in_r_half_arr[i]);
            }

            let noise_strength_g = D::F32Vec::load(d, &noise_strength_g_arr);
            let noise_strength_r = D::F32Vec::load(d, &noise_strength_r_arr);

            // Compute noise contributions (matching C++ lines 221-224)
            let addit_rnd_noise_red = row_rnd_r * norm_const;
            let addit_rnd_noise_green = row_rnd_g * norm_const;
            let addit_rnd_noise_correlated = row_rnd_c * norm_const;

            // Compute red and green noise (matching C++ lines 233-236)
            let red_noise = noise_strength_r * ((k_rgn_corr * addit_rnd_noise_red) + (k_rg_corr * addit_rnd_noise_correlated));
            let green_noise = noise_strength_g * ((k_rgn_corr * addit_rnd_noise_green) + (k_rg_corr * addit_rnd_noise_correlated));

            let rg_noise = red_noise + green_noise;

            // Update channels (matching C++ AddNoiseToRGB)
            let row0 = D::F32Vec::load(d, &row[0][x..]);
            let row1 = D::F32Vec::load(d, &row[1][x..]);
            let row2 = D::F32Vec::load(d, &row[2][x..]);

            let new_row0 = row0 + ((ytox_v * rg_noise) + (red_noise - green_noise));
            let new_row1 = row1 + rg_noise;
            let new_row2 = row2 + (ytob_v * rg_noise);

            new_row0.store(&mut row[0][x..]);
            new_row1.store(&mut row[1][x..]);
            new_row2.store(&mut row[2][x..]);

            x += simd_width;
        }

        // Scalar tail
        for x in x..xsize {
            let row_rnd_r = row[3][x];
            let row_rnd_g = row[4][x];
            let row_rnd_c = row[5][x];
            let vx = row[0][x];
            let vy = row[1][x];
            let in_g = vy - vx;
            let in_r = vy + vx;
            let noise_strength_g = noise.strength(in_g * 0.5);
            let noise_strength_r = noise.strength(in_r * 0.5);
            let addit_rnd_noise_red = row_rnd_r * 0.22;
            let addit_rnd_noise_green = row_rnd_g * 0.22;
            let addit_rnd_noise_correlated = row_rnd_c * 0.22;
            let red_noise = noise_strength_r
                * (0.0078125 * addit_rnd_noise_red + 0.9921875 * addit_rnd_noise_correlated);
            let green_noise = noise_strength_g
                * (0.0078125 * addit_rnd_noise_green + 0.9921875 * addit_rnd_noise_correlated);
            let rg_noise = red_noise + green_noise;
            row[0][x] += ytox * rg_noise + red_noise - green_noise;
            row[1][x] += rg_noise;
            row[2][x] += ytob * rg_noise;
        }
    }
);

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
        _state: Option<&mut dyn std::any::Any>,
    ) {
        let ytox = self.color_correlation.y_to_x_lf();
        let ytob = self.color_correlation.y_to_b_lf();
        add_noise_simd_dispatch(row, xsize, &self.noise, ytox, ytob);
    }
}

#[cfg(test)]
mod test {
    use crate::{
        error::Result,
        features::noise::Noise,
        frame::color_correlation_map::ColorCorrelationParams,
        image::Image,
        render::{
            stages::noise::{AddNoiseStage, ConvolveNoiseStage},
            test::make_and_run_simple_pipeline,
        },
        util::test::assert_almost_abs_eq,
    };
    use test_log::test;

    // TODO(firsching): Add more relevant ConvolveNoise tests as per discussions in https://github.com/libjxl/jxl-rs/pull/60.

    #[test]
    fn convolve_noise_process_row_chunk() -> Result<()> {
        let input: Image<f32> = Image::new_range((2, 2), 0.0, 1.0)?;
        let stage = ConvolveNoiseStage::new(0);
        let output: Vec<Image<f32>> =
            make_and_run_simple_pipeline(stage, &[input], (2, 2), 0, 256)?;
        assert_almost_abs_eq(output[0].row(0)[0], 7.2, 1e-6);
        assert_almost_abs_eq(output[0].row(0)[1], 2.4, 1e-6);
        assert_almost_abs_eq(output[0].row(1)[0], -2.4, 1e-6);
        assert_almost_abs_eq(output[0].row(1)[1], -7.2, 1e-6);
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
            Noise {
                lut: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            },
            ColorCorrelationParams::default(),
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
                    assert_almost_abs_eq(output[c].row(y)[x], want_out[c][y][x], 1e-5);
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
                    Noise {
                        lut: [0.0, 2.0, 1.0, 0.0, 1.0, 3.0, 1.1, 2.3],
                    },
                    ColorCorrelationParams::default(),
                    3,
                )
            },
            (500, 500),
            6,
        )
    }
}

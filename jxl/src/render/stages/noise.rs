// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::{
    features::noise::Noise,
    frame::color_correlation_map::ColorCorrelationParams,
    render::{RenderPipelineInOutStage, RenderPipelineInPlaceStage, RenderPipelineStage},
};

pub struct ConvolveNoiseStage {
    channel: usize,
}

impl ConvolveNoiseStage {
    #[allow(dead_code)]
    pub fn new(channel: usize) -> ConvolveNoiseStage {
        ConvolveNoiseStage { channel }
    }
}

impl std::fmt::Display for ConvolveNoiseStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "convolve noise for channel {}", self.channel,)
    }
}

impl RenderPipelineStage for ConvolveNoiseStage {
    type Type = RenderPipelineInOutStage<f32, f32, 2, 2, 0, 0>;

    fn uses_channel(&self, c: usize) -> bool {
        c == self.channel
    }

    fn process_row_chunk(
        &self,
        _position: (usize, usize),
        xsize: usize,
        row: &mut [(&[&[f32]], &mut [&mut [f32]])],
        _state: Option<&mut dyn std::any::Any>,
    ) {
        let (input, output) = &mut row[0];
        for x in 0..xsize {
            let mut others = 0.0;
            for i in 0..5 {
                let offset = (x as i32 + i) as usize;
                others += input[0][offset];
                others += input[1][offset];
                others += input[3][offset];
                others += input[4][offset];
            }
            others += input[2][x];
            others += input[2][x + 1];
            others += input[2][x + 3];
            others += input[2][x + 4];
            output[0][x] = others * 0.16 + input[2][x + 2] * -3.84;
        }
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

impl RenderPipelineStage for AddNoiseStage {
    type Type = RenderPipelineInPlaceStage<f32>;

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
        let norm_const = 0.22;
        let ytox = self.color_correlation.y_to_x_lf();
        let ytob = self.color_correlation.y_to_b_lf();
        for x in 0..xsize {
            let row_rnd_r = row[3][x];
            let row_rnd_g = row[4][x];
            let row_rnd_c = row[5][x];
            let vx = row[0][x];
            let vy = row[1][x];
            let in_g = vy - vx;
            let in_r = vy + vx;
            let noise_strength_g = self.noise.strength(in_g * 0.5);
            let noise_strength_r = self.noise.strength(in_r * 0.5);
            let addit_rnd_noise_red = row_rnd_r * norm_const;
            let addit_rnd_noise_green = row_rnd_g * norm_const;
            let addit_rnd_noise_correlated = row_rnd_c * norm_const;
            let k_rg_corr = 0.9921875;
            let k_rgn_corr = 0.0078125;
            let red_noise = noise_strength_r
                * (k_rgn_corr * addit_rnd_noise_red + k_rg_corr * addit_rnd_noise_correlated);
            let green_noise = noise_strength_g
                * (k_rgn_corr * addit_rnd_noise_green + k_rg_corr * addit_rnd_noise_correlated);
            let rg_noise = red_noise + green_noise;
            row[0][x] += ytox * rg_noise + red_noise - green_noise;
            row[1][x] += rg_noise;
            row[2][x] += ytob * rg_noise;
        }
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
        util::test::assert_almost_eq,
    };
    use test_log::test;

    // TODO(firsching): Add more relevant ConvolveNoise tests as per discussions in https://github.com/libjxl/jxl-rs/pull/60.

    #[test]
    fn convolve_noise_process_row_chunk() -> Result<()> {
        let input: Image<f32> = Image::new_range((2, 2), 0.0, 1.0)?;
        let stage = ConvolveNoiseStage::new(0);
        let output: Vec<Image<f32>> =
            make_and_run_simple_pipeline(stage, &[input], (2, 2), 0, 256)?.1;
        let rect = output[0].as_rect();
        assert_almost_eq!(rect.row(0)[0], 7.2, 1e-6);
        assert_almost_eq!(rect.row(0)[1], 2.4, 1e-6);
        assert_almost_eq!(rect.row(1)[0], -2.4, 1e-6);
        assert_almost_eq!(rect.row(1)[1], -7.2, 1e-6);
        Ok(())
    }

    #[test]
    fn convolve_noise_consistency() -> Result<()> {
        crate::render::test::test_stage_consistency::<_, f32, f32>(
            ConvolveNoiseStage::new(0),
            (500, 500),
            1,
        )
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
        let output: Vec<Image<f32>> = make_and_run_simple_pipeline(
            stage,
            &[input_c0, input_c1, input_c2, input_c3, input_c4, input_c5],
            (xsize, ysize),
            0,
            256,
        )?
        .1;
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
            let rect = output[c].as_rect();
            for y in 0..rect.size().1 {
                for x in 0..rect.size().0 {
                    assert_almost_eq!(rect.row(y)[x], want_out[c][y][x], 1e-5);
                }
            }
        }
        Ok(())
    }

    #[test]
    fn add_noise_consistency() -> Result<()> {
        crate::render::test::test_stage_consistency::<_, f32, f32>(
            AddNoiseStage::new(
                Noise {
                    lut: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                },
                ColorCorrelationParams::default(),
                3,
            ),
            (500, 500),
            6,
        )
    }
}
